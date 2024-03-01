// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__h__
#define __ivqML__Model__NeuralNetwork__FeedForward__h__

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <ivqML/Model/Base.h>
#include <ivqML/Model/NeuralNetwork/ActivationFactory.h>

namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _S >
      class FeedForward
        : public ivqML::Model::Base< _S >
      {
      public:
        using Self = FeedForward;
        using Superclass = ivqML::Model::Base< _S >;

        using TScalar = typename Superclass::TScalar;
        using TNatural = typename Superclass::TNatural;
        using TMatrix = typename Superclass::TMatrix;
        using TMap = typename Superclass::TMap;

        using TSignature = void( TMap&, const TMap&, bool );
        using TActivation = std::function< TSignature >;
        using TActivationFactory =
          ivqML::Model::NeuralNetwork::ActivationFactory< Self >;

      public:
        FeedForward( );
        virtual ~FeedForward( ) override = default;

        virtual void random_fill( ) override;

        virtual TNatural number_of_inputs( ) const override;
        virtual void set_number_of_inputs( const TNatural& p ) override;
        virtual TNatural number_of_outputs( ) const override;

        virtual void set_number_of_parameters( const TNatural& p ) final;

        void add_layer(
          const TNatural& i, const TNatural& o, const std::string& a
          );
        void add_layer( const TNatural& o, const std::string& a );
        TNatural number_of_layers( ) const;
        void init( );

        template< class _X >
        auto evaluate( const Eigen::EigenBase< _X >& iX ) const
          {
            /* TODO
               TNatural s = 0;
               for( TNatural i = 0; i < this->m_S.size( ); ++i )
               s += this->m_S[ i ];
               std::cout << s << std::endl;
               std::cout << this->number_of_parameters( ) << std::endl;
            */
            TMatrix Z, A;
            unsigned long long L = this->m_W.size( );
            A = iX.derived( ).template cast< TScalar >( );
            for( unsigned long long l = 0; l < L; ++l )
            {
              Z = ( this->m_W[ l ] * A ).colwise( ) + this->m_B[ l ].col( 0 );

              std::cout << "============= " << l << std::endl;
              std::cout << A << std::endl;
              std::cout << "-------------" << std::endl;
              std::cout << Z << std::endl;
              std::cout << "=============" << std::endl;


              TMap mA( A.data( ), A.rows( ), A.cols( ) );
              TMap mZ( Z.data( ), Z.rows( ), Z.cols( ) );
              this->m_F[ l ].second( mA, mZ, false );
            } // end for
            return( A );
          }

        template< class _G, class _X, class _Y >
        void cost(
          Eigen::EigenBase< _G >& iG,
          const Eigen::EigenBase< _X >& iX,
          const Eigen::EigenBase< _Y >& iY,
          TScalar* J = nullptr
          ) const
          {
          }

        template< class _X >
        auto threshold( const Eigen::EigenBase< _X >& iX ) const
          {
            return( iX.derived( ) );
          }

        /* TODO
           virtual void cost(
           TMatrix& G, const TMap& X, const TMap& Y, TScalar* J = nullptr
           ) const override;
           virtual void _evaluate( const TNatural& m ) const override;
        */

      protected:
        virtual void _from_stream( std::istream& i ) override;
        virtual void _to_stream( std::ostream& o ) const override;

      protected:
        bool m_IsLabeling { true }; // TODO: detect this

        std::vector< TNatural > m_S;

        std::vector< TMap > m_W;
        std::vector< TMap > m_B;
        std::vector< std::pair< std::string, TActivation > > m_F;
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/Model/NeuralNetwork/FeedForward.hxx>

#endif // __ivqML__Model__NeuralNetwork__FeedForward__h__

// eof - $RCSfile$
