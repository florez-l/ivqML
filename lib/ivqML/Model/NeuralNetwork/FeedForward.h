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

        template< class _O >
        FeedForward( const _O& other )
          : Superclass( )
          {
          }

        template< class _O >
        Self& operator=( const _O& other )
          {
            this->Superclass::operator=( other );
            return( *this );
          }

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

        template< class _X, class _Y >
        void cost(
          TScalar* bG,
          const Eigen::EigenBase< _X >& iX,
          const Eigen::EigenBase< _Y >& iY,
          TScalar* J = nullptr,
          TScalar* iB = nullptr
          ) const;

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
