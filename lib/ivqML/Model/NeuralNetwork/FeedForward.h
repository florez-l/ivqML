// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Model__NeuralNetwork__FeedForward__h__
#define __ivqML__Model__NeuralNetwork__FeedForward__h__

#include <cstdlib>
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
        auto evaluate(
          const Eigen::EigenBase< _X >& iX,
          TScalar* iB = nullptr
          ) const;

        template< class _X >
        auto threshold( const Eigen::EigenBase< _X >& iX ) const
          {
            return( iX.derived( ) );
          }

        template< class _G, class _X, class _Y >
        void cost(
          Eigen::EigenBase< _G >& iG,
          const Eigen::EigenBase< _X >& iX,
          const Eigen::EigenBase< _Y >& iY,
          TScalar* J = nullptr,
          TScalar* iB = nullptr
          ) const
          {
            using _Gs = typename _G::Scalar;

            // Gradient size
            if( iG.size( ) != this->number_of_parameters( ) )
              iG.derived( ).resize( this->number_of_parameters( ), 1 );
            // TODO: erase this
            std::fill( iG.derived( ).data( ), iG.derived( ).data( ) + iG.size( ), std::numeric_limits< _Gs >::max( ) );

            // Computation buffer
            TNatural m = iX.cols( );
            TNatural bsize = this->m_BSize * m;
            TScalar* buffer = iB;
            if( iB == nullptr )
              buffer =
                reinterpret_cast< TScalar* >(
                  std::malloc( sizeof( TScalar ) * bsize )
                  );

            // Forward propagation
            this->evaluate( iX, buffer );

            // Last layer derivation
            TNatural L = this->number_of_layers( );
            TNatural as = bsize - ( this->m_S[ L ] * m );
            TMap D( buffer + as, this->m_S[ L ], m );
            D -= iY.derived( ).template cast< TScalar >( );

            // Remaining layers
            for( TNatural l = L; l > 0; --l )
            {
            } // end for

            /* TODO
               std::cout << "******************************" << std::endl;
               std::cout << A << std::endl;
               std::cout << "******************************" << std::endl;
            */

            // Free buffer
            if( iB == nullptr )
              std::free( buffer );
          }

      protected:
        virtual void _from_stream( std::istream& i ) override;
        virtual void _to_stream( std::ostream& o ) const override;

      protected:
        bool m_IsLabeling { true }; // TODO: detect this

        std::vector< TNatural > m_S;
        TNatural m_BSize;

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
